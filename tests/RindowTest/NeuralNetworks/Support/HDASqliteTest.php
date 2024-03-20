<?php
namespace RindowTest\NeuralNetworks\Support\HDASqliteTest;

use PHPUnit\Framework\TestCase;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Support\HDA\HDASqlite;
use PDO;

class HDASqliteTest extends TestCase
{
    protected $filename;

    public function setUp() : void
    {
        $this->filename = __DIR__.'/../../../tmp/test.hda.sqlite3';
        $pdo = new PDO('sqlite:'.$this->filename);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $sql = "DROP TABLE IF EXISTS hda";
        $stat = $pdo->exec($sql);
        unset($stat);
        unset($pdo);
    }
    public function fetchAll()
    {
        $pdo = new PDO('sqlite:'.$this->filename);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $sql = "SELECT * FROM hda";
        $stat = $pdo->query($sql);
        foreach ($stat as $value) {
            $rows[] = $value;
        }
        unset($stat);
        unset($pdo);
        return $rows;
    }

    public function testBuilder()
    {
        $nn = new NeuralNetworks();
        $hdaFactory = $nn->utils()->HDA();
        $hda = $hdaFactory->open($this->filename,'a');
        $this->assertInstanceof('Rindow\NeuralNetworks\Support\HDA\HDASqlite',$hda);
    }

    public function testSetGetUnsetRoot()
    {
        $hda = new HDASqlite($this->filename,'a');
        $this->assertFalse(isset($hda['abc']));
        $hda['abc'] = 'def';
        $this->assertEquals('def',$hda['abc']);
        $this->assertTrue(isset($hda['abc']));
        unset($hda['abc']);
        $this->assertFalse(isset($hda['abc']));
    }

    public function testSetGetUnsetDict()
    {
        $hda = new HDASqlite($this->filename,'a');
        $this->assertFalse(isset($hda['abc']));
        $hda['abc'] = ['def'=>'ghi'];
        // replace
        $hda['abc'] = ['wxy'=>'zzz','jkl'=>'www'];
        // tree
        $hda['abc']['wxy'] = ['qqq'=>'ppp'];
        $hda['abcd'] = ['wxy'=>'zzz'];
        $this->assertTrue(isset($hda['abc']));
        $this->assertFalse(isset($hda['abc']['def']));
        $this->assertTrue(isset($hda['abc']['wxy']['qqq']));
        $this->assertTrue(isset($hda['abc']['jkl']));
        unset($hda['abc']);
        $this->assertFalse(isset($hda['abc']));
        $this->assertFalse(isset($hda['abc']['def']));
        $this->assertFalse(isset($hda['abc']['wxy']));
        $this->assertFalse(isset($hda['abc']['wxy']['qqq']));
        $this->assertEquals('zzz',$hda['abcd']['wxy']);
        $this->assertCount(2,$this->fetchAll());
    }

    public function testIterator()
    {
        $hda = new HDASqlite($this->filename,'a');
        $hda['abc'] = 'def';
        $hda['xyz'] = 'www';
        $array = [];
        foreach ($hda as $key => $value) {
            $array[$key] = $value;
        }
        // *** CAUTION ***
        //   Order is not preserved
        ksort($array);
        $this->assertEquals([
            'abc' => 'def',
            'xyz' => 'www',
        ],$array);

        //////
        $hda['def'] = ['aaa','bbb','ccc'];
        $array = [];
        foreach ($hda['def'] as $key => $value) {
            $array[$key] = $value;
        }
        // *** CAUTION ***
        //   Order is not preserved
        ksort($array);
        $this->assertEquals([
            'aaa','bbb','ccc'
        ],$array);
    }
}
